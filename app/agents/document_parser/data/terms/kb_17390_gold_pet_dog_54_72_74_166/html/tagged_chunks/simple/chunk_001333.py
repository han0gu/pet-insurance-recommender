from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>제2조(지급사유)</h1><br><h1 id='201' "
 "style='font-size:14px'>\uf000 회사는 특별약관의</h1><br><p id='202' "
 "data-category='paragraph' style='font-size:14px'>보험기간 중 의료법 제3조에 정한 국내의 종합병원 "
 "또는 이와</p><br><p id='203' data-category='paragraph' "
 "style='font-size:14px'>동등하다고 회사가 인정하는 의료기관에서 전문의 자격증을 가진 자가"),
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
 'indexing': {'chunk_id': 'chunk_001333',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
