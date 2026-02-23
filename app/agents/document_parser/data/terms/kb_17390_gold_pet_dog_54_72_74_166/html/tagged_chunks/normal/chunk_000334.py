from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 71</p><br><p id='187' "
 "data-category='paragraph' style='font-size:18px'>- 71 -</p><p id='188' "
 "data-category='paragraph' style='font-size:20px'>제2절 보통약관의 보장</p><p id='189' "
 "data-category='paragraph' style='font-size:20px'>일반상해80%이상후유장해</p><br><p"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000334',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
