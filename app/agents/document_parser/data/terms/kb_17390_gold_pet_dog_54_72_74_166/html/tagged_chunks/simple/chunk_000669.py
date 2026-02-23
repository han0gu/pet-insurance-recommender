from langchain_core.documents import Document

chunk = Document(
    page_content=("KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><h1 id='223' "
 "style='font-size:20px'>5.</h1><br><h1 id='224' "
 "style='font-size:20px'>호흡기관련질병수술비</h1><br><p id='225' "
 "data-category='paragraph' style='font-size:14px'>제1조(보험금의 지급사유)<br>회사는 피보험자가 "
 '이 특별약관의 보험기간 중에 제3조(호흡기관련질병의 정의 및<br>진단확정)에서 정한 "호흡기관련질병"으로 진단확정되고 그 치료를'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000669',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
