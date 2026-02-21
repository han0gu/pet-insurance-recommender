from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 "보통약관"이라 합니다)을 체결할 때 계약자의 청약과 회사의 승낙으<br>로 보통약관에 부가하여 이루어집니다.<br>\uf000 '
 '이 특별약관을 통하여 전자서명법 제2조 제2호에 따른 전자서명으로 계약을 청약할<br>수 있으며, 이 경우 전자서명은 자필서명과 동일한 '
 "효력을 갖는 것으로 합니다.</p><p id='46' data-category='list' "
 "style='font-size:14px'>제3조(약관교부의</p><br><p id='47' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_001396',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
