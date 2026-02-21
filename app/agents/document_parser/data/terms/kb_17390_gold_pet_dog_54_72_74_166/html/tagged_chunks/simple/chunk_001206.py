from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만,\xa0손해가\xa0그 가족의 고의로</p><br><h1 id='2' style='font-size:14px'>인하여\xa0"
 "발생한 경우에는 그 권리를 취득합니다.</h1><p id='3' data-category='paragraph' "
 "style='font-size:14px'>제16조(계약 전 알릴 의무)<br>계약자, 피보험자 또는 이들의 대리인은 청약할 때 "
 '청약서(질문서를 포함합니다)에<br>서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하 "계약 전<br>알릴 의무"라 '
 '하며, 상법상 "고지의무"와'),
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
 'indexing': {'chunk_id': 'chunk_001206',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
