from langchain_core.documents import Document

chunk = Document(
    page_content=("id='206' style='font-size:16px'>제3조(보험금을 지급하지 않는</h1><br><p id='207' "
 "data-category='paragraph' style='font-size:16px'>보험사고)</p><br><p id='208' "
 "data-category='paragraph' style='font-size:16px'>계약자 또는 지정대리청구인의 고의에 의하여 "
 "피보험자가 제2조(지급사유)의 제1항<br>에 해당된 경우에는 이 특별약관의 보험금을 지급하지 않습니다.</p><h1 id='209'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001339',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
