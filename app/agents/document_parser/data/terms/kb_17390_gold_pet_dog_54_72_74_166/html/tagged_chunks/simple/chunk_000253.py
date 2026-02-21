from langchain_core.documents import Document

chunk = Document(
    page_content=(". 특별</p><br><table id='60' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>용 어 풀 "
 '이</td><td>부활</td><td>약</td></tr><tr><td colspan="3">관 보험료 납입을 연체하여 계약이 해지되고 '
 '계약자가 해약환급금을 받지 않은 경 우에 회사가 정하는 소정의 절차에 따라 해지된 계약을 다시 되살리는 '
 "일</td></tr></tbody></table><p id='61' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000253',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
