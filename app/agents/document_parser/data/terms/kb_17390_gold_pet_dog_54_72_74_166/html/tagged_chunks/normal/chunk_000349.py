from langchain_core.documents import Document

chunk = Document(
    page_content=("사항은 보통약관 제1절 일반조항을 따릅니다.</p><p id='201' data-category='paragraph' "
 "style='font-size:16px'>- 72 -</p><p id='202' data-category='paragraph' "
 "style='font-size:20px'>특별약관</p><p id='203' data-category='paragraph' "
 "style='font-size:16px'>제1장 상해 관련 특별약관</p><p id='204' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000349',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
