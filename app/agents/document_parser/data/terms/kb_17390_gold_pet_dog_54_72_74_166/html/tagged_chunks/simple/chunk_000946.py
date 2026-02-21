from langchain_core.documents import Document

chunk = Document(
    page_content=("70%, 자기부담금 3만원, 기본형Ⅱ 가입 기준</p><br><p id='132' data-category='paragraph' "
 "style='font-size:14px'>예시① 입/통원 중 수술을 하지 않은 날의 경우</p><br><p id='133' "
 "data-category='paragraph' style='font-size:14px'>·피보험자가 부담한 당일 의료비 : "
 "33만원</p><br><p id='134' data-category='paragraph' "
 "style='font-size:14px'>·지급금액 = {(33만원 –"),
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
 'indexing': {'chunk_id': 'chunk_000946',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
