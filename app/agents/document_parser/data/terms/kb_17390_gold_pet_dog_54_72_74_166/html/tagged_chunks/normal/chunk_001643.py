from langchain_core.documents import Document

chunk = Document(
    page_content=("id='173' data-category='paragraph' style='font-size:18px'>- 152 -</p><h1 "
 "id='174' style='font-size:16px'>5) ‘흉복부장기 또는 비뇨생식기 기능에 약간의 장해를 남긴 때’라 "
 '함은<br>아래의 경우 중 하나에 해당하는 때를 말한다.<br>가) 방광의 용량이 50cc 이하로 위축되었거나 요도협착, 배뇨기능 '
 '상실<br>로 영구적인 간헐적 인공요도가 필요한 때<br>나) 음경의 1/2 이상이 결손되었거나 질구 협착으로 성생활이 불가능한 '
 '때<br>다) 폐질환'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001643',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
