from langchain_core.documents import Document

chunk = Document(
    page_content=("및 미경과보험료를 계약자에게 지급합니다.</p><br><p id='198' data-category='paragraph' "
 "style='font-size:14px'>제4조(준용규정)</p><br><p id='199' "
 "data-category='paragraph' style='font-size:14px'>이 보장에서 정하지</p><br><p "
 "id='200' data-category='paragraph' style='font-size:14px'>않은 사항은 보통약관 제1절 "
 "일반조항을 따릅니다.</p><p id='201'"),
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
 'indexing': {'chunk_id': 'chunk_000348',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
