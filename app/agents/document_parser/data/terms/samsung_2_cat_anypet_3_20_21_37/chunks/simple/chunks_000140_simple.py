from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조(보험증권의 발급)\n'
 '① 회사는 계약자에게 보험증권을 드려야 하고, 그 약관의 주요한 내용을 알려드립니다. ② 계약자의 요청이 있을 경우, 개별 피보험자에게는 '
 '가입증명서를 발급하여 드립니다.\n'
 '제7조(적용상의 특칙)\n'
 '계약자가 아닌 단체의 소속원이 보험료 전부 또는 일부를 부담하는 경우에는 그 소속원이 계약자로서 의 권리를 행사할 수 있습니다.\n'
 '제8조(준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 28},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000140',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
