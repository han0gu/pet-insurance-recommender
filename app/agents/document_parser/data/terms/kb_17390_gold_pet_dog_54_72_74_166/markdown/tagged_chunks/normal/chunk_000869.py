from langchain_core.documents import Document

chunk = Document(
    page_content=('을 하여도 더 이상 추상(추한 모습)이 없어지지 않는 경우를 말한다.\n'
 '4) 다발성 반흔(흉터) 발생시 각 판정부위(얼굴, 머리, 목) 내의 다발성 반\n'
 '흔(흉터)의 길이 또는 면적은 합산하여 평가한다. 단, 길이가 5mm 미만\n'
 '의 반흔(흉터)은 합산대상에서 제외한다.\n'
 '5) 추상(추한 모습)이 얼굴과 머리 또는 목 부위에 걸쳐 있는 경우에는 머\n'
 '리 또는 목에 있는 흉터의 길이 또는 면적의 1/2을 얼굴의 추상(추한 모\n'
 '습)으로 보아 산정한다.- \n'
 '144 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)- \n'
 '-'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000869',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
