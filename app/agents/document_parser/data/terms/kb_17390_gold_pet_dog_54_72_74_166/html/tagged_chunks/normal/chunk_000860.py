from langchain_core.documents import Document

chunk = Document(
    page_content=(". 해</p><h1 id='27' style='font-size:16px'>제13조(특별약관 내용의 변경 등)</h1><br><p "
 "id='28' data-category='paragraph' style='font-size:14px'>\uf000 계약자는 회사의 승낙을 "
 '얻어 다음의 사항을 변경할 수 있습니다. 이 경우 승낙을 병<br>서면 등으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.<br>1'),
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
 'indexing': {'chunk_id': 'chunk_000860',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
