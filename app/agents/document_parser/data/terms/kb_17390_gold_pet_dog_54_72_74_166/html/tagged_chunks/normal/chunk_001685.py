from langchain_core.documents import Document

chunk = Document(
    page_content=('지급할 때의 적립이율 계산</td><td></td></tr></thead><tbody><tr><td rowspan="4">구 분 '
 '보장보험금</td><td>적 립 기 간 적</td><td>립 이 율</td></tr><tr><td>지급기일의 다음날부터 30일 이내 '
 '기간</td><td>보험계약대출이율 보험계약대출이율 +</td></tr><tr><td>지급기일의 31일이후부터 60일이내 '
 '기간</td><td>가산이율(4.0%)</td></tr><tr><td>지급기일의 61일이후부터 90일이내 기간 지급기일의 91일이후'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001685',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
