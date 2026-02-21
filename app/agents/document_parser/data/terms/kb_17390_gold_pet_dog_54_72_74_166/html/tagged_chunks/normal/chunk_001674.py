from langchain_core.documents import Document

chunk = Document(
    page_content=("있는 발작 또는 3분 이내에 정상으로 회복되는 발작을 말<br>한다.</p><br><p id='19' "
 "data-category='paragraph' style='font-size:14px'><붙 임></p><br><table id='20' "
 "style='font-size:14px'><thead><tr><td>∙ 일상생활</td><td>기본동작(ADLs) 제한 "
 '장해평가표</td><td>공 통</td></tr></thead><tbody><tr><td rowspan="4">유형 '
 '이동동작</td><td>제한 정도 지급률 1) 특별한'),
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
 'indexing': {'chunk_id': 'chunk_001674',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
