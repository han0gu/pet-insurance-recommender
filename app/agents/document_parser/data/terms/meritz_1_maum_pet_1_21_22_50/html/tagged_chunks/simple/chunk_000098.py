from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약을 체결한 날부터 3년이 지났을 때<br>4. 회사가 이 계약을 청약할 때 반려동물의 건강상태를 판단할 수 있는 '
 '기초자료(건강<br>진단서 사본 등)에 따라 승낙한 경우에 건강진단서 사본 등에 명기되어 있는 사항으<br>로 보험금 지급사유가 '
 '발생하였을 때(계약자 또는 피보험자가 회사에 제출한 기초자<br>료의 내용 중 중요사항을 고의로 사실과 다르게 작성한 때에는 계약을 '
 '해지할 수<br>있습니다)<br>5'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000098',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
