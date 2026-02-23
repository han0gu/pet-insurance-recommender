from langchain_core.documents import Document

chunk = Document(
    page_content=('. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성 또는 그<br>밖의 유해한 특성에 의한 '
 "사고</p><br><table id='66' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>부 가 설 "
 "명</td></tr></tbody></table><br><h1 id='67' style='font-size:16px'>∙ 핵연료물질 : "
 "사용된 연료를 포함합니다.</h1><br><p id='68' data-category='list'"),
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
 'indexing': {'chunk_id': 'chunk_001085',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
