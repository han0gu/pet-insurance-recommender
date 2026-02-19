from langchain_core.documents import Document

chunk = Document(
    page_content=('포괄계약 추가특별약관\n'
 '(단체계약 특별약관에 적용)\n'
 '제1조(적용특칙)\n'
 '① 이 추가특별약관(이하 「특별약관」 이라 합니다)을 첨부한 경우에 보험회사(이하 「회사」 라 합니 다)와 보험계약자(이하 「계약자」 라 '
 '합니다)는 다른 규정에도 불구하고 이 특별약관에 따라 보험 료를 정산합니다. ② 보험료의 정산을 전제로 회사는 보험료 정산 전에 새로이 '
 '증가 또는 교체된 피보험자에 대해 생긴 손해를 보상하여 드립니다.\n'
 '제2조(보험료정산기간)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 38},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000189',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
