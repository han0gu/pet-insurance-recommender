from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 진단시점에 이미 극심한 치매 또는 심한 치매로<br>진행된 경우에는 6개월간 지속적인 치료 후 평가한다.<br>다) 치매의 '
 '장해평가는 전문의(정신건강의학과, 신경과)에 의한 임상치<br>매척도(한국판 Expanded Clinical Dementia '
 "Rating) 검사결과에 따<br>른다.</p><br><p id='11' data-category='paragraph' "
 "style='font-size:16px'>4) 뇌전증</p><p id='12' data-category='list'></p><br><h1 "
 "id='13'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001669',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
