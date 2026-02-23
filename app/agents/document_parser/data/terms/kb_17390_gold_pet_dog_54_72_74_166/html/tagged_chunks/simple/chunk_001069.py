from langchain_core.documents import Document

chunk = Document(
    page_content=(". 검사 및 진단을 위한 수술(생검(生檢), 복강경검사(腹腔鏡檢査) 등)</p><p id='42' "
 "data-category='paragraph' style='font-size:14px'>116 KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01)</p><br><table id='43' "
 "style='font-size:14px'><thead></thead><tbody><tr><td></td></tr><tr><td>용 어 풀 "
 '이 ∙ 절단 : 특정부위를 잘라 내는 것 ∙ 절제 : 특정부위를 잘라 없애는 것 ∙ 흡인 : 주사기 등으로'),
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
 'indexing': {'chunk_id': 'chunk_001069',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
