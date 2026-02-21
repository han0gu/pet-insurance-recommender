from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>·지급금액 = {(33만원 – 3만원) x 70%, 15만원} 중 적은 금액 = "
 "15만원</p><br><h1 id='135' style='font-size:14px'>예시② 입/통원 중 수술을 한 날의 "
 "경우</h1><br><p id='136' data-category='paragraph' "
 "style='font-size:14px'>·피보험자가 부담한 수술 당일 의료비 : 203만원</p><br><p id='137' "
 "data-category='list'"),
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
 'indexing': {'chunk_id': 'chunk_000947',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
