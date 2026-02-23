from langchain_core.documents import Document

chunk = Document(
    page_content=('- 료를 정산합니다.\n'
 '- ② 보험료의 정산을 전제로 회사는 보험료 정산 전에 새로이 증가 또는 교체된 피보험자에 대해 생긴\n'
 '- 손해를 보상하여 드립니다.\n'
 '# 제2조(보험료정산기간)- ① 계약자와 회사는 보험증권에 정한 기간(이하 「정산기간」 이라 합니다)마다 보험료를 정산하기로 합\n'
 '- 니다.\n'
 '- ② 정산기간은 다음 중 어느 하나를 정하여 보험증권에 기재합니다.\n'
 '- - 매월, 매분기, 매반기, 기타()\n'
 '- - 보험기간 종료 후'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000119',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
