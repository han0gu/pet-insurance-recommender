from langchain_core.documents import Document

chunk = Document(
    page_content=("id='99' data-category='paragraph' style='font-size:14px'>됩니다.<br>\uf000 "
 '제2항에도 불구하고, "전환대상계약이 장애인전용보험으로 전환된 해당 연도에 제<br>4조(전환 취소)에 따라 전환을 취소하는 경우"에는 '
 '해당 연도에 납입한 모든 전환<br>대상계약보험료가 보험료 납입영수증에 장애인전용 보장성보험료로 표시되지 않습<br>니다'),
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
 'indexing': {'chunk_id': 'chunk_001431',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
