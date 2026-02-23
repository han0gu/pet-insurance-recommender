from langchain_core.documents import Document

chunk = Document(
    page_content=(". 해</p><br><p id='76' data-category='paragraph' "
 "style='font-size:16px'>∙</p><br><p id='77' data-category='paragraph' "
 "style='font-size:16px'>국세 및 지방세 체납처분 절차</p><br><p id='78' "
 "data-category='paragraph' style='font-size:16px'>국세 및 지방세 체납처분 절차란 국세 또는 "
 '지방세를 체납할 경우 국세 기본법<br>병<br>및 지방세법에 의하여 체납된 세금에 대하여'),
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
 'indexing': {'chunk_id': 'chunk_000902',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
