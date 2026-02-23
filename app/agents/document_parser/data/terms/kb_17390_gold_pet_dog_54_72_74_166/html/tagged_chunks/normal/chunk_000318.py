from langchain_core.documents import Document

chunk = Document(
    page_content=("data-category='list'></p><br><table id='168' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>계약의 청약을 "
 "권유하기</td><td>위하여 만든 자료 등을 말합니다.</td></tr></tbody></table><p id='169' "
 "data-category='paragraph' style='font-size:16px'>- 70 -</p><p id='170' "
 "data-category='list'></p><p id='171'"),
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
 'indexing': {'chunk_id': 'chunk_000318',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
