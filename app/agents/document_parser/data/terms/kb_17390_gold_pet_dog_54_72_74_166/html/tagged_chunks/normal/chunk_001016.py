from langchain_core.documents import Document

chunk = Document(
    page_content=("id='227' data-category='paragraph' style='font-size:16px'>를 받은 경우에는 하나의 "
 '주요치료보험금만 지급하며, 각 치료구분별 보상한도액<br>중 최대 보상한도액에 해당하는 치료에 대하여 제1조(보험금의 지급사유) '
 "제3항<br>에 따라 보상하여 드립니다.</p><br><table id='228' "
 "style='font-size:14px'><thead></thead><tbody><tr><td></td><td>동일한 날에 두 종류 "
 '이상의 "반려동물주요치료"를 받은'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001016',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
