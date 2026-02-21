from langchain_core.documents import Document

chunk = Document(
    page_content=(". 및</td></tr></tbody></table><p id='186' data-category='paragraph' "
 "style='font-size:16px'>제7조(계약 전 알릴 의무)</p><br><p id='187' "
 "data-category='paragraph' style='font-size:14px'>계약자 또는 피보험자는 청약할 때(진단계약의 "
 '경우에는 건강진단할 때를 말합니다)<br>청약서에서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하 "<br>계약 전 '
 '알릴의무"라 하며, 상법상 "고지의무"와'),
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
 'indexing': {'chunk_id': 'chunk_000818',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
