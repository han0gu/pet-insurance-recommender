from langchain_core.documents import Document

chunk = Document(
    page_content=("의료비 – 1일당 자기부담금) X 보상비율}과 1일당</p><br><h1 id='129' "
 "style='font-size:14px'>보상한도액 중 적은 금액</h1><br><p id='130' "
 "data-category='paragraph' style='font-size:14px'>[의료비보험금 지급금액 예시]</p><br><p "
 "id='131' data-category='paragraph' style='font-size:14px'>·보상비율 70%, 자기부담금 "
 "3만원, 기본형Ⅱ 가입 기준</p><br><p id='132'"),
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
 'indexing': {'chunk_id': 'chunk_000945',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
