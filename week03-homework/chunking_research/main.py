from enum import Enum
import json
import os
from pathlib import Path
from re import split
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.core.node_parser import (
    SentenceSplitter,
    SentenceWindowNodeParser,
    TokenTextSplitter,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)
from llama_index.llms.dashscope import DashScope
from llama_index.readers.file import MarkdownReader


load_dotenv(".env")


class SplitterType(Enum):
    """切片方法枚举"""

    SENTENCE = "sentence"
    TOKEN = "token"
    SENTENCE_WINDOW = "sentence_window"

    @property
    def display_name(self) -> str:
        """返回显示名称"""
        display_names = {
            SplitterType.SENTENCE: "句子切片",
            SplitterType.TOKEN: "Token切片",
            SplitterType.SENTENCE_WINDOW: "句子窗口切片",
        }
        return display_names[self]

    @classmethod
    def from_display_name(cls, display_name: str) -> "SplitterType":
        """从显示名称获取枚举值"""
        name_mapping = {
            "句子切片": cls.SENTENCE,
            "Token切片": cls.TOKEN,
            "句子窗口切片": cls.SENTENCE_WINDOW,
        }
        return name_mapping.get(display_name)


def setup_environment():
    """配置LlamaIndex环境和模型"""
    # 设置DashScope API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置DASHSCOPE_API_KEY环境变量")
        print("请设置: export DASHSCOPE_API_KEY='your_api_key'")
        return False

    # 配置LLM
    llm = DashScope(model_name="qwen-plus", api_key=api_key, max_tokens=1024)
    # DashScope(model_name="qwen-turbo", api_key=api_key, max_tokens=1024)

    # 配置嵌入模型
    embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1, api_key=api_key
    )

    # 设置全局配置
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    print("✅ 环境配置完成")
    return True


def load_documents() -> List[Document]:
    """加载测试文档"""
    documents = []
    data_dir = Path(__file__).parent / "data" / "documents"

    parser = MarkdownReader()
    file_extractor = {".md": parser}
    documents = SimpleDirectoryReader(
        data_dir, file_extractor=file_extractor
    ).load_data()

    # for file_path in data_dir.glob("*.md"):
    #     try:
    #         with open(file_path, "r", encoding="utf-8") as f:
    #             content = f.read()

    #         # 过滤掉导航和界面元素，只保留实际内容
    #         lines = content.split("\n")
    #         filtered_lines = []

    #         for line in lines:
    #             # 跳过维基百科的导航、链接等界面元素
    #             if any(
    #                 skip_word in line.lower()
    #                 for skip_word in [
    #                     "wikipedia",
    #                     "维基百科",
    #                     "编辑",
    #                     "讨论",
    #                     "查看",
    #                     "工具",
    #                     "accesskey",
    #                     "hreflang",
    #                     'class="',
    #                     'href="',
    #                     "![",
    #                     "移至侧栏",
    #                     "隐藏",
    #                     "外观",
    #                     "打印",
    #                     "下载",
    #                 ]
    #             ):
    #                 continue

    #             # 保留有实际内容的行
    #             if (
    #                 len(line.strip()) > 10
    #                 and not line.startswith("|")
    #                 and not line.startswith("+")
    #             ):
    #                 filtered_lines.append(line)

    #         filtered_content = "\n".join(filtered_lines)

    #         # 只有当过滤后的内容足够长时才添加
    #         if len(filtered_content) > 500:
    #             doc = Document(
    #                 text=filtered_content,
    #                 metadata={"filename": file_path.name, "source": str(file_path)},
    #             )
    #             documents.append(doc)
    #             print(
    #                 f"✅ 加载文档: {file_path.name} (长度: {len(filtered_content)}字符)"
    #             )

    #     except Exception as e:
    #         print(f"❌ 加载文档失败 {file_path.name}: {e}")

    print(f"📚 总共加载 {len(documents)} 个文档")
    return documents


def create_splitter(splitter_type: SplitterType, **kwargs):
    """根据枚举类型创建对应的切片器"""
    if splitter_type == SplitterType.SENTENCE:
        return create_sentence_splitter(**kwargs)
    elif splitter_type == SplitterType.TOKEN:
        return create_token_splitter(**kwargs)
    elif splitter_type == SplitterType.SENTENCE_WINDOW:
        return create_sentence_window_splitter(**kwargs)
    else:
        raise ValueError(f"不支持的切片器类型: {splitter_type}")


def create_sentence_splitter(
    chunk_size: int = 512, chunk_overlap: int = 50
) -> SentenceSplitter:
    """创建句子切片器"""
    # 限制chunk_size不超过嵌入模型限制
    max_chunk_size = 2000  # 留一些余量
    if chunk_size > max_chunk_size:
        print(f"警告: chunk_size {chunk_size} 超过限制，调整为 {max_chunk_size}")
        chunk_size = max_chunk_size

    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
    )


def create_token_splitter(
    chunk_size: int = 512, chunk_overlap: int = 50
) -> TokenTextSplitter:
    """创建Token切片器"""
    # 限制chunk_size不超过嵌入模型限制
    max_chunk_size = 2000  # 留一些余量
    if chunk_size > max_chunk_size:
        print(f"警告: chunk_size {chunk_size} 超过限制，调整为 {max_chunk_size}")
        chunk_size = max_chunk_size

    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
    )


def create_sentence_window_splitter(
    window_size: int = 3, window_metadata_key: str = "window"
) -> SentenceWindowNodeParser:
    """创建句子窗口切片器"""
    # 需要先创建一个基础的句子切片器
    sentence_splitter = SentenceSplitter(
        chunk_size=1024,  # 设置较大的chunk_size
        chunk_overlap=20,
    )

    return SentenceWindowNodeParser(window_size=window_size)


def validate_and_clean_nodes(nodes_list, splitter_type, max_length=2048, min_length=1):
    """验证并清理节点，确保文本长度符合嵌入模型要求

    Args:
        nodes_list: 节点列表
        splitter_type: 切片器类型
        max_length: 最大文本长度限制
        min_length: 最小文本长度限制

    Returns:
        List[TextNode]: 清理后的有效节点列表
    """
    cleaned_nodes = []

    for i, node in enumerate(nodes_list):
        try:
            text = node.text.strip() if hasattr(node, "text") else str(node)

            # 检查文本长度
            if len(text) < min_length:
                print(f"跳过节点 {i}: 文本太短 (长度: {len(text)})")
                continue

            if len(text) > max_length:
                # 截断过长的文本
                original_length = len(text)
                text = text[:max_length]
                print(f"截断节点 {i}: {original_length} -> {len(text)} 字符")

                # 更新节点文本
                if hasattr(node, "text"):
                    node.text = text
                else:
                    # 如果不是标准节点，创建新的TextNode
                    from llama_index.core.schema import TextNode

                    node = TextNode(
                        text=text,
                        id_=f"{splitter_type.value}_truncated_node_{i}",
                        metadata=getattr(node, "metadata", {}),
                    )

            cleaned_nodes.append(node)

        except Exception as e:
            print(f"处理节点 {i} 时出错: {e}")
            continue

    return cleaned_nodes


def evaluate_splitter(
    documents: List[Document],
    splitter,
    splitter_type: SplitterType,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> Dict[str, Any]:
    """评估切片方法的效果"""
    splitter_name = splitter_type.display_name
    print(f"\n🔍 评估切片方法: {splitter_name}")

    # 根据切片器类型显示不同的参数信息
    if splitter_type == SplitterType.SENTENCE_WINDOW:
        print(f"参数: window_size={getattr(splitter, 'window_size', 3)}")
    else:
        print(f"参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    start_time = time.time()

    try:
        # 创建节点
        if splitter_type == SplitterType.SENTENCE_WINDOW:
            # 为SentenceWindowNodeParser添加特殊处理
            print("Processing documents for SentenceWindowNodeParser...")
            for i, doc in enumerate(documents):
                print(f"Document {i}: type={type(doc)}, has_id={hasattr(doc, 'id_')}")
                if not hasattr(doc, "id_") or not doc.id_:
                    doc.id_ = f"doc_{i}"

        nodes = splitter.get_nodes_from_documents(documents)

        # 关键修复：检查nodes是否为None
        if nodes is None:
            print(f"❌ 错误: {splitter_name} 返回了 None")
            return {
                "splitter_name": splitter_name,
                "splitter_type": splitter_type.value,
                "error": "Splitter returned None - check documents and splitter configuration",
                "processing_time": time.time() - start_time,
                "node_count": 0,
                "avg_node_length": 0,
                "responses": [],
            }

        # 检查nodes是否为空列表
        if not isinstance(nodes, list):
            print(f"❌ 错误: {splitter_name} 返回了 {type(nodes)}，期望列表")
            return {
                "splitter_name": splitter_name,
                "splitter_type": splitter_type.value,
                "error": f"Splitter returned {type(nodes)}, expected list",
                "processing_time": time.time() - start_time,
                "node_count": 0,
                "avg_node_length": 0,
                "responses": [],
            }

        print(f"原始节点数量: {len(nodes)}")
        print(f"第一个节点类型: {type(nodes[0]) if nodes else 'None'}")

        # 验证节点格式并修复
        valid_nodes = []
        for i, node in enumerate(nodes):
            if isinstance(node, str):
                # 如果是字符串，创建一个新的Node对象
                from llama_index.core.schema import TextNode

                new_node = TextNode(
                    text=node,
                    id_=f"{splitter_type.value}_node_{i}",
                    metadata={"source": f"document_{i // 10}", "chunk_id": i},
                )
                valid_nodes.append(new_node)
            elif hasattr(node, "text") and hasattr(node, "id_"):
                # 如果是正确的Node对象，确保有id_
                if not node.id_:
                    node.id_ = f"{splitter_type.value}_node_{i}"
                valid_nodes.append(node)
            else:
                # 处理其他类型的节点
                print(f"警告: 节点 {i} 类型异常: {type(node)}")
                try:
                    # 尝试获取节点的文本内容
                    if hasattr(node, "text"):
                        text_content = node.text
                    elif hasattr(node, "get_content"):
                        text_content = node.get_content()
                    else:
                        text_content = (
                            str(node) if hasattr(node, "__str__") else f"node_{i}"
                        )

                    new_node = TextNode(
                        text=text_content,
                        id_=f"{splitter_type.value}_node_{i}",
                        metadata={"source": f"document_{i // 10}", "chunk_id": i},
                    )
                    valid_nodes.append(new_node)
                except Exception as e:
                    print(f"无法处理节点 {i}: {e}")
                    continue

        nodes = valid_nodes

        # 应用节点验证和清理（调用模块级函数）
        print(f"验证节点文本长度...")
        nodes = validate_and_clean_nodes(nodes, splitter_type)

        if not nodes:
            print(f"⚠️  警告: 验证后没有有效节点")
            return {
                "splitter_name": splitter_name,
                "splitter_type": splitter_type.value,
                "error": "No valid nodes after text length validation",
                "processing_time": time.time() - start_time,
                "node_count": 0,
                "avg_node_length": 0,
                "responses": [],
            }

        # 打印节点长度统计
        text_lengths = [len(node.text) for node in nodes if hasattr(node, "text")]
        if text_lengths:
            print(
                f"节点文本长度统计: 最小={min(text_lengths)}, 最大={max(text_lengths)}, 平均={sum(text_lengths) / len(text_lengths):.1f}"
            )

        print(f"✅ 验证完成，有效节点数量: {len(nodes)}")

        # 构建索引时添加错误处理
        try:
            # 最终验证：确保传递给VectorStoreIndex的是有效的节点列表
            if not isinstance(nodes, list) or not nodes:
                raise ValueError(
                    f"Invalid nodes for VectorStoreIndex: type={type(nodes)}, length={len(nodes) if hasattr(nodes, '__len__') else 'unknown'}"
                )

            # 验证每个节点都是有效的TextNode且文本长度合适
            for i, node in enumerate(nodes):
                if not hasattr(node, "text") or not hasattr(node, "id_"):
                    raise ValueError(
                        f"Invalid node at index {i}: missing text or id_ attribute"
                    )
                if not (1 <= len(node.text) <= 2048):
                    raise ValueError(
                        f"Invalid text length at node {i}: {len(node.text)} (should be 1-2048)"
                    )

            print(f"创建VectorStoreIndex，节点数量: {len(nodes)}")
            index = VectorStoreIndex(nodes)
        except Exception as e:
            print(f"❌ 构建索引失败: {str(e)}")
            return {
                "splitter_name": splitter_name,
                "splitter_type": splitter_type.value,
                "error": f"Index creation failed: {str(e)}",
                "processing_time": time.time() - start_time,
                "node_count": len(nodes) if nodes else 0,
                "avg_node_length": sum(len(node.text) for node in nodes) / len(nodes)
                if nodes
                else 0,
                "responses": [],
            }

        # 创建查询引擎
        if splitter_type == SplitterType.SENTENCE_WINDOW:
            # 句子窗口切片需要特殊的后处理器
            postprocessor = MetadataReplacementPostProcessor(
                target_metadata_key="window"
            )
            query_engine = index.as_query_engine(
                node_postprocessors=[postprocessor], similarity_top_k=3
            )
        else:
            query_engine = index.as_query_engine(similarity_top_k=3)

        # 测试查询
        test_queries = [
            "什么是大语言模型？",
            "RAG技术的核心思想是什么？",
            "SCP基金会是什么组织？",
        ]

        responses = []
        for query in test_queries:
            try:
                response = query_engine.query(query)
                responses.append(
                    {
                        "query": query,
                        "response": str(response),
                        "source_nodes": len(response.source_nodes)
                        if hasattr(response, "source_nodes")
                        else 0,
                    }
                )
            except Exception as e:
                print(f"查询失败: {query} - {e}")
                responses.append(
                    {"query": query, "response": f"查询失败: {e}", "source_nodes": 0}
                )

        processing_time = time.time() - start_time

        # 计算统计信息
        node_count = len(nodes)
        avg_node_length = (
            sum(len(node.text) for node in nodes) / node_count if node_count > 0 else 0
        )

        results = {
            "splitter_name": splitter_name,
            "splitter_type": splitter_type.value,
            "parameters": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            "statistics": {
                "node_count": node_count,
                "avg_node_length": round(avg_node_length, 2),
                "processing_time": round(processing_time, 2),
            },
            "test_results": responses,
        }

        print(f"✅ 节点数量: {node_count}")
        print(f"✅ 平均节点长度: {avg_node_length:.2f}字符")
        print(f"✅ 处理时间: {processing_time:.2f}秒")

        return results

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return {
            "splitter_name": splitter_name,
            "splitter_type": splitter_type.value,
            "error": str(e),
            "parameters": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        }


def run_parameter_comparison(documents: List[Document]):
    """运行参数对比实验"""
    print("\n🧪 开始参数对比实验")

    # 测试不同的参数组合
    parameter_combinations = [
        {"chunk_size": 256, "chunk_overlap": 25},
        {"chunk_size": 512, "chunk_overlap": 50},
        {"chunk_size": 1024, "chunk_overlap": 100},
        {"chunk_size": 512, "chunk_overlap": 0},  # 无重叠
        {"chunk_size": 512, "chunk_overlap": 128},  # 高重叠
    ]

    all_results = []

    for params in parameter_combinations:
        print(f"\n📊 测试参数组合: {params}")

        # 测试句子切片
        sentence_splitter = create_splitter(SplitterType.SENTENCE, **params)
        sentence_results = evaluate_splitter(
            documents, sentence_splitter, SplitterType.SENTENCE, **params
        )
        all_results.append(sentence_results)

        # 测试Token切片
        token_splitter = create_splitter(SplitterType.TOKEN, **params)
        token_results = evaluate_splitter(
            documents, token_splitter, SplitterType.TOKEN, **params
        )
        all_results.append(token_results)

    # 测试句子窗口切片（使用默认参数）
    window_splitter = create_splitter(SplitterType.SENTENCE_WINDOW)
    window_results = evaluate_splitter(
        documents, window_splitter, SplitterType.SENTENCE_WINDOW
    )
    all_results.append(window_results)

    return all_results


def save_results(results: List[Dict], output_file: str = "experiment_results.json"):
    """保存实验结果"""
    output_path = Path(__file__).parent / output_file

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 实验结果已保存到: {output_path}")
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")


def print_summary(results: List[Dict]):
    """打印实验结果摘要"""
    print("\n📋 实验结果摘要")
    print("=" * 60)

    for result in results:
        if "error" in result:
            print(f"❌ {result['splitter_name']}: {result['error']}")
            continue

        stats = result.get("statistics", {})
        params = result.get("parameters", {})
        splitter_type = result.get("splitter_type", "unknown")

        print(f"\n🔸 {result['splitter_name']} ({splitter_type})")
        print(
            f"   参数: chunk_size={params.get('chunk_size', 'N/A')}, chunk_overlap={params.get('chunk_overlap', 'N/A')}"
        )
        print(f"   节点数: {stats.get('node_count', 'N/A')}")
        print(f"   平均长度: {stats.get('avg_node_length', 'N/A')}字符")
        print(f"   处理时间: {stats.get('processing_time', 'N/A')}秒")


def main():
    """主函数"""
    print("🚀 开始执行文本切片对比实验")

    # 1. 配置环境
    if not setup_environment():
        print("❌ 环境配置失败，请检查DASHSCOPE_API_KEY")
        return

    # 2. 加载文档
    documents = load_documents()
    if not documents:
        print("❌ 没有找到有效的文档")
        return

    # 3. 运行参数对比实验
    results = run_parameter_comparison(documents)

    # 4. 保存和展示结果
    save_results(results)
    print_summary(results)

    print("\n✅ 实验完成！请查看experiment_results.json文件获取详细结果")
    print("📝 接下来请完善report.md文件中的实验分析")


if __name__ == "__main__":
    main()
