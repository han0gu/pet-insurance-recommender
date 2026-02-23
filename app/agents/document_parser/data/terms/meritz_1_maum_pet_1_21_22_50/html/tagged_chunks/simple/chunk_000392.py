from langchain_core.documents import Document

chunk = Document(
    page_content=("제9조 제4항 관련)</p><table id='83' style='font-size:14px'><thead><tr><td>구 "
 '분</td><td>기 간</td><td>지 급 이 자</td></tr></thead><tbody><tr><td '
 'rowspan="4">보장관련 보험금</td><td>지급기일의 다음 날부터 30일 이내 '
 '기간</td><td>보험계약대출이율</td></tr><tr><td>지급기일의 31일이후부터 60일이내 '
 '기간</td><td>보험계약대출이율+ 가산이율(4.0%)</td></tr><tr><td>지급기일의'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000392',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
