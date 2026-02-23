from langchain_core.documents import Document

chunk = Document(
    page_content=("70%, 자기부담금 3만원, 기본형Ⅱ 가입 기준</p><br><p id='204' data-category='paragraph' "
 "style='font-size:14px'>·MRI/CT</p><br><h1 id='205' style='font-size:14px'>시행 "
 "시 보상한도액 : 100만원 한도</h1><br><p id='206' data-category='list' "
 "style='font-size:14px'>예시①<br>·MRI/CT 시행 당일 피보험자가 부담한 의료비 : "
 '78만원<br>·반려동물의료비보험금 :'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000999',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
