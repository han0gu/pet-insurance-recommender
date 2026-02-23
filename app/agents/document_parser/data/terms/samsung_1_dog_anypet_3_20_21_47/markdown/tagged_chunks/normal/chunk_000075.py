from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 계약자 또는 피보험자가 보험금 청구에 관한 서류에 고의로 사실과 다른 것을 기재하였거나 그\n'
 '- 서류 또는 증거를 위조 또는 변조한 경우. 다만, 이미 보험금 지급사유가 발생한 경우에는 보험\n'
 '- 금 지급에 영향을 미치지 않습니다.\n'
 '【설명】 계약자, 피보험자 또는 보험수익자가 보험금 청구에 관한 서류에 고의로 사실과 다른 것을 기\n'
 '재하였거나 그 서류 또는 증거를 위조 또는 변조한 경우 회사는 그 사실을 안 날부터 1개월 이내에 계'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000075',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
