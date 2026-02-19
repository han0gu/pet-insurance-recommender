from langchain_core.documents import Document

chunk = Document(
    page_content=('- 매월, 매분기, 매반기, 기타() - 보험기간 종료 후\n'
 '제3조(예치보험료)\n'
 '① 회사는 보험기간 중에 가입할 것으로 예상하는 피보험자의 수에 기초하여 연간 예상되는 보험료(이 하 「예상보험료」 라 합니다)를 '
 '계산합니다. 다만, 달리 약정한 경우에는 정산기간 단위로 예상보 험료를 계산할 수도 있습니다. ② 계약자는 계약할 때에 제1항의 '
 '예상보험료를 예치보험료로 회사에 납입해야 합니다. 다만, 제1항의 단서의 경우에는 해당 정산기간이 시작되기 전에 해당 예치보험료를 회사에 '
 '납입해야 합니다.\n'
 '제4조(보험료의 정산)'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 30},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000148',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
