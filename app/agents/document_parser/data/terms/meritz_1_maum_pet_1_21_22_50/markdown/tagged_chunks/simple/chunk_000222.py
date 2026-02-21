from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 지급기일의 다음 날부터 30일 이내 기간 | 보험계약대출이율 |\n'
 '| 지급기일의 31일이후부터 60일이내 기간 | 보험계약대출이율+ 가산이율(4.0%) |\n'
 '| 지급기일의 61일이후부터 90일이내 기간 | 보험계약대출이율+ 가산이율(6.0%) |\n'
 '| 지급기일의 91일이후 기간 | 보험계약대출이율+ 가산이율(8.0%) |\n'
 '주) 보험계약대출이율은 보험개발원이 공시하는 보험계약대출이율을 적용합니다.- 49 -<부표3> 단기요율표| 보 험 기 간 | 단 기 요 '
 '율 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000222',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
