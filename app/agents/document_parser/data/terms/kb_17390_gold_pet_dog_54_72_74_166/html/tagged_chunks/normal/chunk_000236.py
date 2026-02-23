from langchain_core.documents import Document

chunk = Document(
    page_content=("id='45' style='font-size:14px'>용 어 풀 이 자동대출납입</h1><br><table id='46' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>보험료를 제때에 "
 '납입하기</td><td>곤란한 경우에</td><td>계약자가 자동대출납입을 신청하면</td></tr><tr><td '
 'colspan="3">해당 보험 상품의 해약환급금 범위 내에서 납입할 보험료를 자동적으로 대출하 여 이를 보험료 납입에 충당하는 서비스를'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000236',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
