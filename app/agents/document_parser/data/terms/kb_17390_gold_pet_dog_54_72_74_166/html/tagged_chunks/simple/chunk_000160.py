from langchain_core.documents import Document

chunk = Document(
    page_content=("불구하고 청약한 날부터 30일(단, 만 65세 이상의 계약자가</p><br><p id='199' "
 "data-category='paragraph' style='font-size:14px'>통신수단 중</p><p id='200' "
 "data-category='paragraph' style='font-size:18px'>- 62 -</p><p id='201' "
 "data-category='list' style='font-size:16px'>전화를 이용하여 체결한 경우 45일)이 초과된 계약은 "
 '청약을 철회할 수 없습니다.<br>\uf000 청약철회는'),
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
 'indexing': {'chunk_id': 'chunk_000160',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
