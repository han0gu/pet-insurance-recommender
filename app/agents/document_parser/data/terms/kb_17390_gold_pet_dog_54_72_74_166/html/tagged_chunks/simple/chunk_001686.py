from langchain_core.documents import Document

chunk = Document(
    page_content=('61일이후부터 90일이내 기간 지급기일의 91일이후 기간</td><td>보험계약대출이율 + 가산이율(6.0%) 보험계약대출이율 + '
 '가산이율(8.0%)</td></tr><tr><td rowspan="2">만기환급금 및 해약환급금</td><td>지급사유가 발생한 날의 '
 '다음날부터 청구일까지의 기간</td><td>1년이내 : 공시이율의 50% 1년초과기간 : 공시이율의 '
 '40%</td></tr><tr><td>청구일의 다음 날부터 지급일까지의 '
 '기간</td><td>보험계약대출이율</td></tr></tbody></table><br><table'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001686',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
