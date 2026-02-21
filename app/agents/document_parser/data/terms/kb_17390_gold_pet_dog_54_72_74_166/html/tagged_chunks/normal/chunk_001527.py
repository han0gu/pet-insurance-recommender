from langchain_core.documents import Document

chunk = Document(
    page_content=("발치한 정상치아, 노화로 인해 자연 발치된 치아, 보철</p><br><p id='220' data-category='paragraph' "
 "style='font-size:14px'>KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 143</p><br><p id='221' "
 "data-category='paragraph' style='font-size:18px'>- 143 -</p><br><p id='222' "
 "data-category='list'></p><header id='0' style='font-size:14px'>(복합레진,"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_001527',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
