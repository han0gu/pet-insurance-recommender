from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>관</p><table id='0' "
 "style='font-size:14px'><thead></thead><tbody><tr><td></td></tr><tr><td>관 련 법 "
 '규 전자서명법 제2조(정의) "전자서명" 이란 다음 각 목의 사항을 나타내는 데 이용하기 위하여 전자문서 에 첨부되거나 논리적으로 결합된 '
 '전자적형태의 정보를 말한다'),
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
 'indexing': {'chunk_id': 'chunk_000184',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
