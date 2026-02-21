from langchain_core.documents import Document

chunk = Document(
    page_content=("수의사에 의해 발급한 것이어야 합니다.</h1><br><table id='163' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>관 련 법 규</td><td>수의사법 "
 '제12조(진단서 등)</td></tr><tr><td colspan="2">① 수의사는 자기가 직접 진료하거나 검안하지 아니하고는 진단서, '
 '검안서, 증 명서 또는 처방전을 발급하지 못하며, 「약사법」 제85조제6항에 따른 동물 용 의약품(이하 "동물용 의약품"이라 한다)을 '
 '처방·투약하지 못한다'),
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
 'indexing': {'chunk_id': 'chunk_000797',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
