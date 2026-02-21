from langchain_core.documents import Document

chunk = Document(
    page_content=("적용할 경우 보험증권에 그 내용을 기재하여 드립니다.</p><br><p id='101' "
 "data-category='list'></p><br><p id='102' data-category='paragraph' "
 "style='font-size:14px'>제8조(준용규정)</p><br><h1 id='103' "
 "style='font-size:14px'>이</h1><br><p id='104' data-category='paragraph' "
 "style='font-size:14px'>특별약관에서 정하지 않은 사항은 보통약관 제1절 일반조항을"),
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
 'indexing': {'chunk_id': 'chunk_001271',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
