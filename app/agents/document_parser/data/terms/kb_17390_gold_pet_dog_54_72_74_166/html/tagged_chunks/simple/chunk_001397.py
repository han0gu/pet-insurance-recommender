from langchain_core.documents import Document

chunk = Document(
    page_content=("id='47' data-category='paragraph' style='font-size:14px'>특례)</p><br><p "
 "id='48' data-category='paragraph' style='font-size:14px'>\uf000 계약자가 원하는 방법에 "
 "따라 상품설명서, 보험약관 및 계약자 보관용 청약서 등</p><br><p id='49' data-category='list' "
 'style=\'font-size:14px\'>(보험증권은 제외하며, 이하 "보험계약 안내자료"라 합니다)을 전자우편 및 전자<br>적 '
 '의사표시로 제공한 경우,'),
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
 'indexing': {'chunk_id': 'chunk_001397',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
