from langchain_core.documents import Document

chunk = Document(
    page_content=(". 전자서명법 제2조 제2호에 따른 전자서명을 포<br>약<br>함합니다.</p><p id='219' "
 "data-category='paragraph' style='font-size:16px'>용 어 풀 이 약관의 중요한 "
 "내용</p><br><p id='220' data-category='paragraph' style='font-size:16px'>금융소비자 "
 "보호에 관한 법률 제19조(설명의무)등에서 정한 다음의 내용을 말<br>합니다.</p><br><p id='221' "
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
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000177',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
