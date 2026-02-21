from langchain_core.documents import Document

chunk = Document(
    page_content=('| 구 분 보장보험금 | 지급기일의 31일이후부터 60일이내 기간 | 가산이율(4.0%) |\n'
 '| 구 분 보장보험금 | 지급기일의 61일이후부터 90일이내 기간 지급기일의 91일이후 기간 | 보험계약대출이율 + 가산이율(6.0%) '
 '보험계약대출이율 + 가산이율(8.0%) |\n'
 '| 만기환급금 및 해약환급금 | 지급사유가 발생한 날의 다음날부터 청구일까지의 기간 | 1년이내 : 공시이율의 50% 1년초과기간 : '
 '공시이율의 40% |\n'
 '| 만기환급금 및 해약환급금 | 청구일의 다음 날부터 지급일까지의 기간 | 보험계약대출이율 |'),
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
 'indexing': {'chunk_id': 'chunk_000953',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
