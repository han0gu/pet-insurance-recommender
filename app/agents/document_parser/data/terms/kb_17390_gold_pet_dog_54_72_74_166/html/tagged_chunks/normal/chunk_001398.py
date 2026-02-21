from langchain_core.documents import Document

chunk = Document(
    page_content=('"보험계약 안내자료"라 합니다)을 전자우편 및 전자<br>적 의사표시로 제공한 경우, 계약자 또는 그 대리인이 보험계약 안내자료를 '
 '수신<br>하였을 때에는 해당 문서를 드린 것으로 봅니다.<br>\uf000 계약자가 보험계약 안내자료에 대하여 전자적 방법의 수령을 '
 '원하지 않는 경우에<br>는 청약한 날로부터 5영업일 이내에 보험계약 안내자료를 우편 등의 방법으로 계<br>약자에게 '
 "드립니다.</p><br><p id='50' data-category='paragraph' "
 "style='font-size:14px'>제4조(계약자의 알릴"),
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
 'indexing': {'chunk_id': 'chunk_001398',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
