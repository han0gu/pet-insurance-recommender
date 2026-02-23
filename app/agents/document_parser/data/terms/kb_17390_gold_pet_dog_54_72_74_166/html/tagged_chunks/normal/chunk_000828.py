from langchain_core.documents import Document

chunk = Document(
    page_content=('. 위 이외에 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때<br>\uf000 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 '
 "경우에는 보통약관 제1절</p><br><h1 id='3' style='font-size:14px'>일반조항 제22조(계약내용의 변경 "
 "등)에 따라 계약내용을 변경할 수 있습니다.</h1><br><p id='4' data-category='paragraph' "
 "style='font-size:14px'>부 가 설 명 위험변경에 따른 계약 변경 절차</p><br><figure id='5'><img"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000828',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
