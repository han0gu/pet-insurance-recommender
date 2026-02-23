from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>예 시</p><br><p id='126' data-category='paragraph' "
 "style='font-size:14px'>의료비보험금의 계산</p><br><h1 id='127' "
 "style='font-size:14px'>[의료비보험금 산출방식]</h1><br><p id='128' "
 "data-category='paragraph' style='font-size:14px'>{(피보험자가 부담한 1일당 의료비 – 1일당 "
 "자기부담금) X 보상비율}과 1일당</p><br><h1 id='129'"),
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
 'indexing': {'chunk_id': 'chunk_000944',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
