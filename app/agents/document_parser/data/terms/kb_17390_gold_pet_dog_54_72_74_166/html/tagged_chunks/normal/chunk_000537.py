from langchain_core.documents import Document

chunk = Document(
    page_content=("id='287' data-category='paragraph' style='font-size:22px'>상해</p><br><p "
 "id='288' data-category='list'></p><p id='289' data-category='list'></p><p "
 "id='0' data-category='paragraph' style='font-size:14px'>및 급여 상대가치점수」의 개정에 따라 "
 '제1항의 "수가코드"가 폐지 또는 변경되</p><br><p id=\'1\' data-category=\'list\''),
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
 'indexing': {'chunk_id': 'chunk_000537',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
