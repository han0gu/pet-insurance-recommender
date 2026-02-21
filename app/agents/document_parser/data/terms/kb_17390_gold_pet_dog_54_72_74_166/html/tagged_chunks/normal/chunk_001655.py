from langchain_core.documents import Document

chunk = Document(
    page_content=(". 그러나, 12개<br>월이 지났다고 하더라도 뚜렷하게 기능 향상이 진행되고 있는 경우</p><br><p id='187' "
 "data-category='list'></p><br><p id='188' data-category='paragraph' "
 "style='font-size:14px'>표</p><br><p id='189' data-category='paragraph' "
 "style='font-size:14px'>별</p><p id='190' data-category='paragraph' "
 "style='font-size:16px'>KB"),
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
 'indexing': {'chunk_id': 'chunk_001655',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
