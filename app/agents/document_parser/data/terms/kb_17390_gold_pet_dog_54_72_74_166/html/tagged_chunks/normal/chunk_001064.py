from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 직접 진료하거나 검안한 수의사가 부득이한 사유로 진단서, 검안서 또는 증명서를 발급할 수 없을 때에는 같은 동물병원에 종사하는 '
 '다른 수의사가 진료부 등에 의하여 발급할 수 있다. ② 제1항에 따른 진료 중 폐사(斃死)한 경우에 발급하는 폐사 진단서는 다른 '
 '수의사에게서 발급받을 수 있다. ③ 수의사는 직접 진료하거나 검안한 동물에 대한 진단서, 검안서, 증명서 또는 처방전의 발급을 요구받았을 '
 '때에는 정당한 사유 없이 이를 거부하여서는 아니된다'),
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
 'indexing': {'chunk_id': 'chunk_001064',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
