from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유 판정에 드는</p><br><h1 id='151' "
 "style='font-size:14px'>의료비용은 회사가 전액 부담합니다.</h1><h1 id='152' "
 "style='font-size:14px'>제3조(보험금을 지급하지 않는 사유)</h1><br><p id='153' "
 "data-category='paragraph' style='font-size:14px'>\uf000 회사는 아래의 사유로 인한 손해는 "
 "보상하지 않습니다.</p><br><p id='154'"),
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
 'indexing': {'chunk_id': 'chunk_000960',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
