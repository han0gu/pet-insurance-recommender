from langchain_core.documents import Document

chunk = Document(
    page_content=(". 손해의 방지 또는 경감을 위하여 노력하는 일(피해자에 대한 응급처치, 긴급</p><p id='201' "
 "data-category='paragraph' style='font-size:18px'>- 122 -</p><p id='202' "
 "data-category='list' style='font-size:16px'>호송 또는 그 밖의 긴급조치를 포함합니다)<br>2. "
 '제3자로부터 손해의 배상을 받을 수 있는 경우에는 그 권리를 지키거나 행사<br>하기 위한 필요한 조치를 취하는 일<br>3'),
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
 'indexing': {'chunk_id': 'chunk_001182',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
