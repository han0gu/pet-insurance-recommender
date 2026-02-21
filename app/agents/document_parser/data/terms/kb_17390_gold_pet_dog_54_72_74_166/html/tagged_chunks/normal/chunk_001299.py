from langchain_core.documents import Document

chunk = Document(
    page_content=('포함합니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각<br>각 산출한 보상책임액의 합계액이 손해액을 초과할 때에는 '
 "아래에 따라 손해를<br>보상합니다.</p><br><p id='138' data-category='paragraph' "
 "style='font-size:14px'>다른 계약이 없을 때</p><h1 id='139' "
 "style='font-size:14px'>\uf000</h1><br><figure id='140'><img "
 'style=\'font-size:14px\' alt="피보험자가 이 계약의 지급보험금'),
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
 'indexing': {'chunk_id': 'chunk_001299',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
