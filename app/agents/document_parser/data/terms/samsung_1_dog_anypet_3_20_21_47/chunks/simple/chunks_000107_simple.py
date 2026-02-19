from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 일반금융소비자에게 청약을 권유하거나 일반금융소비자가 설명을 요청하는 경우 보험상품 에 관한 중요한 사항을 계약자가 이해할 수 '
 '있도록 설명하고 계약자가 이해하였음을 서명( 「전자 서명법」 제2조 제2호에 따른 전자서명을 포함), 기명날인 또는 녹취 등을 통해 '
 '확인받아야 하며, 설명서를 제공하여야 합니다. ② 설명서, 약관, 청약서 부본 및 증권의 제공 사실에 관하여 계약자와 회사간에 다툼이 '
 '있는 경우에 는 회사가 이를 증명하여야 합니다'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 19},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000107',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
